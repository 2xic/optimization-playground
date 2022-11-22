#include <stdlib.h>
#include <stdio.h>
#include "shared.h"
#include <string.h>

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

int getElement(Matrix *a, int row, int col)
{
    int rowIndex = a->columns * row;
    return a->data[rowIndex + col];
}

void setElement(Matrix *a, int row, int col, int value)
{
    int rowIndex = a->columns * row;
    a->data[rowIndex + col] = value;
}

void setElementN(Matrix *a, int location[], int value)
{
    size_t inputSize = getSize(location);
    size_t arrSize = getSize(a->size);
    if (inputSize != arrSize){
        return;
    }
    int tensorLocation = 0;
    for (int i = 0; i < arrSize; i++) {
        if (0 == i) {
            tensorLocation = location[i];
        } else {
            tensorLocation = tensorLocation * a->size[i] + location[i];
        }
        printf("tensor == %i %i %i\n", i, tensorLocation, a->size[i]);
    }
    printf("%i\n", tensorLocation);
    a->data[tensorLocation] = value;
}

int getSize(int size[]){
    size_t arrSize = sizeof(size)/sizeof(size[0]);
    return arrSize;
}

void freeMatrix(Matrix *a)
{
    printf("freeing the matreix\n");
    free(a->data);
    free(a->size);
    free(a);
}

// TODO: this should support any dimension.
Matrix *
createMatrix(int rows, int columns)
{
    Matrix *matrix = malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        return NULL;
    }
    matrix->rows = rows;
    matrix->columns = columns;

    int size = rows * columns * sizeof(int *);
    matrix->data = (int *)malloc(size);
    printf("should fill ? %i (%i, %i)\n", size, rows, columns);
    memset(matrix->data, 0, size);
    printf("memset :)\n");

    return matrix;
}


void fill(Matrix *a, int value) {
    int size = a->rows * a->columns * sizeof(int *);
    printf("filling %i (%i, %i) \n", size, a->rows, a->columns);
    for (int i = 0; i < size; i++) {
        a->data[i] = value;
    }
}

Matrix *createMatrixN(int size[])
{
    Matrix *matrix = malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        return NULL;
    }
    matrix->size = (int *)malloc(sizeof(size));
    for(int i = 0; i < getSize(matrix->size); i++) {
        matrix->size[i] = size[i];
    }

    int nSize = 0;
    size_t arrSize = getSize(size);
    for (int i = 0; i < arrSize; i++) {
        printf("size[i] == %i\n", size[i]);

        if( i== 0) {
            nSize = size[i];
            matrix->columns = size[i];
        } else {
            matrix->rows = size[i];
            nSize *= size[i];
        }
    }

    int dataSize = nSize * sizeof(int *);
    printf("dataSize = %i\n", dataSize);

    matrix->data = (int *)malloc(dataSize);
    memset(matrix->data, 0, dataSize);

    return matrix;
}

Matrix *MatMul(Matrix *a, Matrix *b)
{
    // printf("(%d, %d) (%d, %d)\n", a->rows, a->columns, b->rows, b->columns);
    // printf("%d is equal\n", (a->columns != b->rows));

    if (a->columns != b->rows)
    {
        printf("Not equal dimensions\n");
        return NULL;
    }
    Matrix *results = createMatrix(
        a->rows,
        b->columns);

    for (int row = 0; row < results->rows; row++)
    {
        for (int column = 0; column < results->columns; column++)
        {
            int accumulator = 0;
            for (int column_j = 0; column_j < results->columns; column_j++)
            {
                // printf("%d * %d, \n", getElement(a, row, column_j), getElement(b, column_j, column) );
                accumulator += getElement(a, row, column_j) * getElement(b, column_j, column);
            }
            // printf("%d acc\n", accumulator);

            setElement(
                results,
                row,
                column,
                accumulator);
        }
        // printf("\n");
    }

    return results;
}
