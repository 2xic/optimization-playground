#include <stdlib.h>
#include <stdio.h>
#include "shared.h"
#include <string.h>

void print_array(float *ptr, int length)
{
    // printf("[");
    for (int i = 0; i < length; i++)
    {
        if (i > 0)
        {
            // printf(", ");
        }
        // printf("%f", ptr[i]);
    }
    // printf("]");
}


int getElement( Matrix *a, int row, int col) {
    int rowIndex = a->columns * row;
    return a->data[rowIndex + col];
}

void setElement( Matrix *a, int row, int col, int value) {
    int rowIndex = a->columns * row;
    a->data[rowIndex + col] = value;
}

void freeMatrix( Matrix *a) {
    free(a->data);
    free(a);
}



// TODO: this should support any dimension.
Matrix *
createMatrix(int rows, int columns)
{
    Matrix *matrix = malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->columns = columns;

    int size = rows * columns * sizeof(int*);
    matrix->data = (int*) malloc(size);
    memset(matrix->data, 0, size);

    return matrix;
}

 Matrix *MatMul( Matrix *a,  Matrix *b)
{
    // printf("(%d, %d) (%d, %d)\n", a->rows, a->columns, b->rows, b->columns);
    // printf("%d is equal\n", (a->columns != b->rows));

    if (a->columns != b->rows){
        // printf("Not equal\n");
        return NULL;
    }
    Matrix *results = createMatrix(
        a->rows,
        b->columns
    );

    

    for(int row = 0; row < results->rows; row ++) {
        int accumulator = 0;
        for(int column= 0; column < results->columns; column ++) {
            int accumulator = 0;
            for(int column_j= 0; column_j < results->columns; column_j ++) {
                // printf("%d * %d, \n", getElement(a, row, column_j), getElement(b, column_j, column) );
                accumulator +=  getElement(a, row, column_j) * getElement(b, column_j, column);
            }   
            // printf("%d acc\n", accumulator);

            setElement(
                results,
                row,
                column,
                accumulator
            );
        }
        // printf("\n");
    }
    

    return results;
}
