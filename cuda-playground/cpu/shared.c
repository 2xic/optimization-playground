#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "shared.h"
#include "extensions.c"


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

float getElement(Matrix *a, int row, int col)
{
    int rowIndex = a->columns * row;
    return a->data[rowIndex + col];
}

void setElement(Matrix *a, int row, int col, float value)
{
    int rowIndex = a->columns * row;
    a->data[rowIndex + col] = value;
}

void setElementN(Matrix *a, int *location, int length, float value)
{
    size_t inputSize = length; // getSize(location);
    size_t arrSize = length;   // getSize(a->size);
    if (inputSize != arrSize)
    {
        return;
    }
    int tensorLocation = 0;
    for (int i = 0; i < arrSize; i++)
    {
        int locI = *(location + i);
        if (0 == i)
        {
            tensorLocation = locI;
        }
        else
        {
            tensorLocation = tensorLocation * a->size[i] + locI;
        }
        printf("tensor == %i %i %i\n", i, tensorLocation, a->size[i]);
    }
    printf("%i\n", tensorLocation);
    a->data[tensorLocation] = value;
}

int isEqual(Matrix *a, Matrix *b)
{
    if (a->rows != b->rows || b->columns != a->columns)
    {
        printf("Not equal dimensions\n");
        return -1;
    }

    int isEqual = 1;

    for (int row = 0; row < a->rows; row++)
    {
        for (int column = 0; column < a->columns; column++)
        {
            isEqual = (getElement(a, row, column) == getElement(b, row, column));
            if (!isEqual)
            {
                printf("%f != %f \n", getElement(a, row, column), getElement(b, row, column));
                break;
            }
        }
    }

    return isEqual;
}

void freeMatrix(Matrix *a)
{
    printf("Freeing the matrix\n");
    free(a->data);
    if (a->size != NULL)
    {
        free(a->size);
    }
    free(a);
}

// TODO: this should support any dimension.
Matrix *createMatrix(int rows, int columns)
{
    Matrix *matrix = malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        return NULL;
    }
    matrix->rows = rows;
    matrix->columns = columns;
    matrix->size = NULL;
    matrix->device = 0;

    int size = rows * columns * sizeof(float *);
    matrix->data = (float *)malloc(size);
    printf("should fill ? %i (%i, %i)\n", size, rows, columns);
    memset(matrix->data, 0, size);
    printf("memset :)\n");

    return matrix;
}

void fillRandom(Matrix *a)
{
    int size = a->rows * a->columns; // * sizeof(int *);
    printf("filling randomly %i (%i, %i) \n", size, a->rows, a->columns);
    for (int i = 0; i < size; i++)
    {
        float random = 1 * ((float)rand()) / (float)RAND_MAX;
        printf("value %f\n", random);
        a->data[i] = random;
    }
}

void fill(Matrix *a, int value)
{
    int size = a->rows * a->columns; // * sizeof(int *);
    printf("filling %i (%i, %i) \n", size, a->rows, a->columns);
    for (int i = 0; i < size; i++)
    {
        a->data[i] = value;
    }
}

Matrix *createMatrixN(int size[], int length)
{
    Matrix *matrix = malloc(sizeof(Matrix));
    if (matrix == NULL)
    {
        return NULL;
    }
    matrix->size = (int *)malloc(sizeof(int *) * length);
    for (int i = 0; i < length; i++)
    {
        matrix->size[i] = size[i];
    }

    int nSize = 0;
    int arrSize = length; // getSize(size);
    printf("length size %i\n", arrSize);
    for (int i = 0; i < arrSize; i++)
    {
        printf("\tsize[i] == %i\n", size[i]);

        if (i == 0)
        {
            nSize = size[i];
            matrix->rows = size[i];
        }
        else
        {
            if (i == 1)
            {
                matrix->columns = size[i];
            }
            nSize *= size[i];
        }
    }

    int dataSize = nSize * sizeof(int *);
    printf("dataSize = %i\n", dataSize);

    matrix->device = 0;
    matrix->data = (float *)malloc(dataSize);
    memset(matrix->data, 0, dataSize);

    return matrix;
}

Matrix *ApplyOperator(Matrix *a, Matrix *b, float *constant, int direction, int operator)
{
    Matrix *c = createMatrix(a->rows, a->columns);

    for (int i = 0; i < a->columns; i++)
    {
        for (int j = 0; j < a->rows; j++)
        {
            float (*getter)(int i, int j) = create_subtract_extension(b, constant);


            float value = getter(i, j);

            float new_value = 0;
            if (direction == 0)
            {
                new_value = create_operator_extension(getElement(a, i, j), value, operator)();
            }
            else
            {
                new_value = create_operator_extension(value, getElement(a, i, j), operator)();
            }

            setElement(c, i, j, new_value);
        }
    }
    return c;
}

Matrix *Add(Matrix *a, Matrix *b)
{
      return ApplyOperator(
        a,
        b,
        NULL,
        0,
        ADD);
}

Matrix *AddConstant(Matrix *a, float b, int direction)
{
      return ApplyOperator(
        a,
        NULL,
        &b,
        direction,
        ADD);
}

Matrix *Subtract(Matrix *a, Matrix *b)
{
    return ApplyOperator(
        a,
        b,
        NULL,
        0,
        SUBTRACT);
}

Matrix *SubtractConstant(Matrix *a, float b, int direction)
{
    return ApplyOperator(
        a,
        NULL,
        &b,
        direction,
        SUBTRACT);
}

Matrix *Mul(Matrix *a, Matrix *b)
{
    return ApplyOperator(
        a,
        b,
        NULL,
        0,
        MUL);
}

Matrix *MulConstant(Matrix *a, float b, int direction)
{
    return ApplyOperator(
        a,
        NULL,
        &b,
        direction,
        MUL);
}

Matrix *DivideConstant(Matrix *a, float b, int direction)
{
    return ApplyOperator(
        a,
        NULL,
        &b,
        direction,
        DIV);
}

Matrix *Exp(Matrix *a)
{
    printf("EXP:)\n");
    Matrix *c = createMatrix(a->rows, a->columns);
    for (int i = 0; i < a->columns; i++)
    {
        for (int j = 0; j < a->rows; j++)
        {
            setElement(c, i, j,
                       exp(getElement(a, i, j)));
        }
    }
    return c;
}

Matrix *MatMul(Matrix *a, Matrix *b)
{
    printf("(%d, %d) @ (%d, %d)\n", a->rows, a->columns, b->rows, b->columns);
    // printf("%d is equal\n", (a->columns != b->rows));

    if (a->columns != b->rows)
    {
        printf("Not equal dimensions\n");
        return NULL;
    }
    Matrix *results = createMatrix(
        a->rows,
        b->columns);

    for (int row = 0; row < a->rows; row++)
    {
        for (int column = 0; column < b->columns; column++)
        {
            float accumulator = 0;
            for (int column_j = 0; column_j < b->rows; column_j++)
            {
                accumulator += getElement(a, row, column_j) * getElement(b, column_j, column);
            }
            setElement(
                results,
                row,
                column,
                accumulator);
        }
    }

    return results;
}
