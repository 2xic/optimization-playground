#ifndef shared_H_   
#define shared_H_


typedef struct {
    int *data;
    int rows;
    int columns;
} Matrix;

int getElement(Matrix *a, int row, int col);

void setElement(Matrix *a, int row, int col, int value);

void freeMatrix(Matrix *a);

Matrix *createMatrix(int rows, int columns);

Matrix *MatMul(Matrix *a, Matrix *b);

#endif
